package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class SetTableAllignment {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置表格对齐方式.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.BOLD);// 实例化字体
            PdfPTable table = new PdfPTable(3);// 定义表格
            table.setTotalWidth(200);// 设置表格宽度为200
            table.setLockedWidth(true);
            PdfPCell cell = new PdfPCell(new Paragraph("new table colspan 3"));// 定义一个表格单元
            cell.setColspan(3);// 设置表格跨度
            table.addCell(cell);// 将单元加入到表格
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("1.1");
            table.addCell("2.1");
            table.addCell("3.1");
            table.setHorizontalAlignment(Element.ALIGN_LEFT);// 设置水平对齐方式 居左
            
            PdfPTable table1 = new PdfPTable(3);// 定义表格
            table1.setTotalWidth(200);// 设置表格宽度为200
            table1.setLockedWidth(true);
            PdfPCell cell2 = new PdfPCell(new Paragraph("new table colspan 3"));// 定义一个表格单元
            cell2.setColspan(3);// 设置表格跨度
            table1.addCell(cell2);// 将单元加入到表格
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table1.addCell("1.1");
            table1.addCell("2.1");
            table1.addCell("3.1");
            table1.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置水平对齐方式 居中
            PdfPTable table2 = new PdfPTable(3);// 定义表格
            table2.setTotalWidth(200);// 设置表格宽度为200
            table2.setLockedWidth(true);
            PdfPCell cell3 = new PdfPCell(new Paragraph("new table colspan 3"));// 定义一个表格单元
            cell3.setColspan(3);// 设置表格跨度
            table2.addCell(cell2);// 将单元加入到表格
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table2.addCell("1.1");
            table2.addCell("2.1");
            table2.addCell("3.1");
            table2.setHorizontalAlignment(Element.ALIGN_RIGHT);// 设置水平对齐方式 居右
            Paragraph p = new Paragraph("表格居左对齐\n\n", FontChinese);
            p.setAlignment(Element.ALIGN_LEFT);
            document.add(p);// 向文档添加内容
            document.add(table);// 将表格添加到文档
            Paragraph p2 = new Paragraph("表格居中对齐\n\n", FontChinese);
            p2.setAlignment(Element.ALIGN_CENTER);
            document.add(p2);// 向文档添加内容
            document.add(table1);// 将表格添加到文档
            Paragraph p3 = new Paragraph("表格居右对齐\n\n", FontChinese);
            p3.setAlignment(Element.ALIGN_RIGHT);
            document.add(p3);// 向文档添加内容
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
