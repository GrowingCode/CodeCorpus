package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Font;
import com.lowagie.text.Table;
import com.lowagie.text.pdf.BaseFont;
import com.lowagie.text.pdf.PdfWriter;

public class CreateTableColumn {
    
    public static void main(String[] args) {
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\创建具有指定列数的表格.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont bfChinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基本字体
            Font fontChinese = new Font(bfChinese, 28, Font.NORMAL);// 实例化字体
            document.add(new com.lowagie.text.Paragraph("这是一个具有5列的表格",
                    fontChinese));// 向文档中添加内容
            Table table = new Table(5);// 创建一个5列的表格
            // 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("1,1");
            table.addCell("1,2");
            table.addCell("1,3");
            table.addCell("1,4");
            table.addCell("1,5");
            table.addCell("2,1");
            table.addCell("2,2");
            table.addCell("2,3");
            table.addCell("2,4");
            table.addCell("2,5");
            document.add(table);// 将表格添加到文档中
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
