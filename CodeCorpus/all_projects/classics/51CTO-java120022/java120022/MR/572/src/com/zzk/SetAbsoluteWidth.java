package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetAbsoluteWidth {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置绝对宽度.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.NORMAL);// 实例化字体
            PdfPTable table1 = new PdfPTable(3);// 定义表格
            PdfPCell cell1 = new PdfPCell(new Paragraph("new table colspan 3"));// 定义一个表格单元
            cell1.setColspan(3);// 设置表格跨度
            table1.addCell(cell1);// 将单元加入到表格
            table1.addCell("1.1");// 将单元格顺次的加入到表格，当一行充满时自动换行
            table1.addCell("2.1");
            table1.addCell("3.1");
            table1.addCell("1.2");
            table1.addCell("2.2");
            table1.addCell("3.2");
            table1.addCell("1.3");
            table1.addCell("2.3");
            table1.addCell("3.3");
            PdfPTable table2 = new PdfPTable(3);// 定义表格
            PdfPCell cell2 = new PdfPCell(new Paragraph("new table colspan 3"));// 定义一个表格单元
            cell2.setColspan(3);// 设置表格跨度
            table2.addCell(cell2);// 将单元加入到表格
            table2.addCell("1.1");// 将单元格顺次的加入到表格，当一行充满时自动换行
            table2.addCell("2.1");
            table2.addCell("3.1");
            table2.addCell("1.2");
            table2.addCell("2.2");
            table2.addCell("3.2");
            table2.addCell("1.3");
            table2.addCell("2.3");
            table2.addCell("3.3");
            table2.setTotalWidth(200);// 设置表格宽度为200
            table2.setLockedWidth(true);// 锁定宽度
            document.add(new Paragraph("                  默认的表格\n\n", FontChinese));
            document.add(table1);// 将表格添加到文档
            document.add(new Paragraph("                  设置表格宽度200",
                    FontChinese));
            document.add(table2);// 将表格添加到文档
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
