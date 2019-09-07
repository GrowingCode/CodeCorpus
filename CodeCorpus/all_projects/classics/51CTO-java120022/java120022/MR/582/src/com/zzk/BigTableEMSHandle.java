package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class BigTableEMSHandle {
    
    public static void main(String args[]) {
        int bigtablesize = 5;
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "c:\\大表格的内存处理.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            
            BaseFont chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font fontChinese1 = new Font(chinese, 10, Font.NORMAL,
                    new BaseColor(90, 90, 90));// 实例化字体类与设置字体大小属性
            Font fontChinese2 = new Font(chinese, 15, Font.NORMAL,
                    BaseColor.BLUE);// 实例化字体类与设置字体大小属性
            document.add(new Paragraph("大表格的内存管理\n\n", fontChinese2));
            float[] hw = { 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.3f };// 设置列宽
            PdfPTable table = new PdfPTable(hw);// 创建表格
            table.setHeaderRows(2);// 设置头排
            table.addCell("10%");// 将单元格内容顺次的加入到表格，当一行充满时自动换行
            table.addCell("20%");
            table.addCell("10%");
            table.addCell("20%");
            table.addCell("10%");
            table.addCell("30%");
            for (int i = 1; i <= 500; i++) {// 循环向表格中添加500条记录
                if (i % bigtablesize == 4) {// 求余
                    document.add(table);// 向文档添加表格
                    table.deleteBodyRows();// 删除多余行
                    table.setSkipFirstHeader(true);// 使表头始终保持在首行
                }
                PdfPCell cell0 = new PdfPCell(new Paragraph(String.valueOf(i),
                        fontChinese1));// 向单元格添加内容
                table.addCell(cell0);// 向表格添加单元格
                PdfPCell cell1 = new PdfPCell(new Paragraph("明日科技",
                        fontChinese1));// 向单元格添加内容
                table.addCell(cell1);// 向表格添加单元格
                PdfPCell cell2 = new PdfPCell(new Paragraph(String.valueOf(i),
                        fontChinese1));// 向单元格添加内容
                table.addCell(cell2);// 向表格添加单元格
                PdfPCell cell3 = new PdfPCell(new Paragraph("明日科技",
                        fontChinese1));// 向单元格添加内容
                table.addCell(cell3);// 向表格添加单元格
            }
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
